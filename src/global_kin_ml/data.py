from __future__ import annotations

import importlib.util
import re
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


def _power_metadata_from_path(raw_path: Path) -> dict[str, object]:
    match = re.search(r"(\d+(?:\.\d+)?)mJ", raw_path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not determine power from filename: {raw_path.name}")
    power_mj = float(match.group(1))
    return {
        "source_file": raw_path.name,
        "power_label": f"{match.group(1)}mJ",
        "power_mj": power_mj,
    }


def _merge_cases_from_directory(module: ModuleType, raw_dir: Path) -> tuple[list[object], dict[str, object], pd.DataFrame]:
    raw_files = sorted(raw_dir.glob("*.out"))
    if not raw_files:
        raise FileNotFoundError(f"No .out files found in {raw_dir}")

    all_cases: list[object] = []
    case_metadata_rows: list[dict[str, object]] = []
    file_summary_rows: list[dict[str, object]] = []
    input_labels: list[str] | None = None
    reaction_labels: list[str] | None = None
    power_species_labels: list[str] | None = None
    unique_local_case_ids: set[int] = set()
    unique_e_over_n_values: set[float] = set()

    global_case_offset = 0
    density_group_offset = 0

    for source_file_id, raw_file in enumerate(raw_files, start=1):
        file_cases, file_metadata = module.parse_dataset(raw_file)
        power_metadata = _power_metadata_from_path(raw_file)

        if input_labels is None:
            input_labels = list(file_metadata["input_species_labels"])
            reaction_labels = list(file_metadata["reaction_labels"])
            power_species_labels = list(file_metadata["power_species_labels"])
        else:
            if list(file_metadata["input_species_labels"]) != input_labels:
                raise ValueError(f"Input species labels changed in {raw_file.name}")
            if list(file_metadata["reaction_labels"]) != reaction_labels:
                raise ValueError(f"Reaction labels changed in {raw_file.name}")
            if list(file_metadata["power_species_labels"]) != power_species_labels:
                raise ValueError(f"Power species labels changed in {raw_file.name}")

        file_summary_rows.append(
            {
                "source_file_id": source_file_id,
                "source_file": raw_file.name,
                "power_label": power_metadata["power_label"],
                "power_mj": power_metadata["power_mj"],
                "file_case_count": file_metadata["total_cases"],
                "file_density_group_count": file_metadata["density_group_count"],
                "file_local_case_count": len(file_metadata["unique_local_case_ids"]),
                "file_unique_e_over_n_count": len(file_metadata["unique_e_over_n_values"]),
            }
        )

        for case in file_cases:
            file_case_id = case.global_case_id
            density_group_in_file_id = case.density_group_id
            unique_local_case_ids.add(case.local_case_id)
            unique_e_over_n_values.add(case.e_over_n_v_cm2)

            case.global_case_id = global_case_offset + file_case_id
            case.density_group_id = density_group_offset + density_group_in_file_id

            case_metadata_rows.append(
                {
                    "global_case_id": case.global_case_id,
                    "source_file_id": source_file_id,
                    "source_file": raw_file.name,
                    "power_group_id": source_file_id,
                    "power_label": power_metadata["power_label"],
                    "power_mj": power_metadata["power_mj"],
                    "file_case_id": file_case_id,
                    "density_group_id": case.density_group_id,
                    "density_group_in_file_id": density_group_in_file_id,
                    "local_case_id": case.local_case_id,
                }
            )
            all_cases.append(case)

        global_case_offset += file_metadata["total_cases"]
        density_group_offset += file_metadata["density_group_count"]

    merged_metadata = {
        "source_path": str(raw_dir),
        "source_type": "directory",
        "source_file_count": len(raw_files),
        "total_cases": len(all_cases),
        "density_group_count": density_group_offset,
        "power_level_count": len(raw_files),
        "power_labels": [row["power_label"] for row in file_summary_rows],
        "power_values_mj": [row["power_mj"] for row in file_summary_rows],
        "input_species_labels": input_labels or [],
        "reaction_labels": reaction_labels or [],
        "power_species_labels": power_species_labels or [],
        "unique_local_case_ids": sorted(unique_local_case_ids),
        "unique_e_over_n_values": sorted(unique_e_over_n_values),
        "case_block_lines": getattr(module, "CASE_BLOCK_LINES", 344),
        "per_file_summary": file_summary_rows,
    }
    return all_cases, merged_metadata, pd.DataFrame(case_metadata_rows)


def _write_generalized_parser_outputs(
    module: ModuleType,
    cases: list[object],
    metadata: dict[str, object],
    case_metadata: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    species_map = pd.DataFrame(module.build_species_map_rows(metadata["input_species_labels"]))
    reaction_map = pd.DataFrame(module.build_reaction_map_rows(metadata["reaction_labels"]))
    case_features = pd.DataFrame(module.build_case_feature_rows(cases)).merge(
        case_metadata,
        on=["global_case_id", "density_group_id", "local_case_id"],
        how="left",
    )
    input_long = pd.DataFrame(module.build_input_long_rows(cases)).merge(
        case_metadata,
        on=["global_case_id", "density_group_id", "local_case_id"],
        how="left",
    )
    power_long = pd.DataFrame(module.build_power_long_rows(cases)).merge(
        case_metadata,
        on=["global_case_id", "density_group_id", "local_case_id"],
        how="left",
    )
    rate_long = pd.DataFrame(module.build_rate_long_rows(cases)).merge(
        case_metadata,
        on=["global_case_id", "density_group_id", "local_case_id"],
        how="left",
    )
    training_inputs = pd.DataFrame(module.build_training_input_rows(cases)).merge(
        case_metadata,
        on=["global_case_id", "density_group_id", "local_case_id"],
        how="left",
    )
    training_targets = pd.DataFrame(module.build_training_target_rows(cases)).merge(
        case_metadata,
        on=["global_case_id", "density_group_id", "local_case_id"],
        how="left",
    )

    leading_input_columns = [
        "global_case_id",
        "source_file_id",
        "source_file",
        "power_group_id",
        "power_label",
        "power_mj",
        "file_case_id",
        "density_group_id",
        "density_group_in_file_id",
        "local_case_id",
        "e_over_n_v_cm2",
    ]
    input_feature_columns = [column for column in training_inputs.columns if column.startswith("input_")]
    training_inputs = training_inputs[leading_input_columns + input_feature_columns]

    leading_target_columns = [
        "global_case_id",
        "source_file_id",
        "source_file",
        "power_group_id",
        "power_label",
        "power_mj",
        "file_case_id",
        "density_group_id",
        "density_group_in_file_id",
        "local_case_id",
    ]
    target_columns = [column for column in training_targets.columns if column.startswith("rate_const_")]
    training_targets = training_targets[leading_target_columns + target_columns]

    parser_summary_rows = [
        {"metric": "source_path", "value": metadata["source_path"]},
        {"metric": "source_type", "value": metadata["source_type"]},
        {"metric": "source_file_count", "value": metadata["source_file_count"]},
        {"metric": "total_cases", "value": metadata["total_cases"]},
        {"metric": "density_group_count", "value": metadata["density_group_count"]},
        {"metric": "power_level_count", "value": metadata["power_level_count"]},
        {"metric": "power_labels", "value": "|".join(str(value) for value in metadata["power_labels"])},
        {"metric": "power_values_mj", "value": "|".join(str(value) for value in metadata["power_values_mj"])},
        {"metric": "local_case_count_observed", "value": len(metadata["unique_local_case_ids"])},
        {"metric": "unique_e_over_n_values", "value": len(metadata["unique_e_over_n_values"])},
        {"metric": "case_block_lines", "value": metadata["case_block_lines"]},
        {"metric": "input_species_count", "value": len(metadata["input_species_labels"])},
        {"metric": "rate_count", "value": len(metadata["reaction_labels"])},
        {"metric": "power_species_count", "value": len(metadata["power_species_labels"])},
        {
            "metric": "min_cases_per_density_group",
            "value": int(training_inputs.groupby("density_group_id").size().min()),
        },
        {
            "metric": "max_cases_per_density_group",
            "value": int(training_inputs.groupby("density_group_id").size().max()),
        },
        {
            "metric": "cases_per_power_level_min",
            "value": int(training_inputs.groupby("power_label").size().min()),
        },
        {
            "metric": "cases_per_power_level_max",
            "value": int(training_inputs.groupby("power_label").size().max()),
        },
    ]
    parser_summary = pd.DataFrame(parser_summary_rows)
    per_file_summary = pd.DataFrame(metadata["per_file_summary"])

    outputs = {
        "parser_summary": output_dir / "parser_summary.csv",
        "parser_file_summary": output_dir / "parser_file_summary.csv",
        "case_features": output_dir / "case_features.csv",
        "species_map": output_dir / "species_map.csv",
        "reaction_map": output_dir / "reaction_map.csv",
        "input_mole_fractions_long": output_dir / "input_mole_fractions_long.csv",
        "power_deposition_long": output_dir / "power_deposition_long.csv",
        "rate_constants_long": output_dir / "rate_constants_long.csv",
        "training_inputs": output_dir / "training_inputs.csv",
        "training_targets": output_dir / "training_targets.csv",
    }
    parser_summary.to_csv(outputs["parser_summary"], index=False)
    per_file_summary.to_csv(outputs["parser_file_summary"], index=False)
    case_features.to_csv(outputs["case_features"], index=False)
    species_map.to_csv(outputs["species_map"], index=False)
    reaction_map.to_csv(outputs["reaction_map"], index=False)
    input_long.to_csv(outputs["input_mole_fractions_long"], index=False)
    power_long.to_csv(outputs["power_deposition_long"], index=False)
    rate_long.to_csv(outputs["rate_constants_long"], index=False)
    training_inputs.to_csv(outputs["training_inputs"], index=False)
    training_targets.to_csv(outputs["training_targets"], index=False)
    return outputs


def parse_raw_dataset(raw_path: Path, output_dir: Path) -> dict[str, Path]:
    module = _load_legacy_parser_module()
    if raw_path.is_dir():
        cases, metadata, case_metadata = _merge_cases_from_directory(module, raw_path)
        return _write_generalized_parser_outputs(
            module=module,
            cases=cases,
            metadata=metadata,
            case_metadata=case_metadata,
            output_dir=output_dir,
        )
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
    parser_summary = {
        str(row["metric"]): row["value"] for _, row in dataset.parser_summary.iterrows()
    }
    expected_total_cases = int(float(parser_summary.get("total_cases", input_rows)))
    expected_density_groups = int(float(parser_summary.get("density_group_count", density_groups)))
    expected_local_case_count = int(float(parser_summary.get("local_case_count_observed", e_over_n_values)))
    expected_input_species = int(float(parser_summary.get("input_species_count", len(dataset.species_map))))
    expected_rate_count = int(float(parser_summary.get("rate_count", len(dataset.reaction_map))))
    expected_input_cols = len(
        [column for column in dataset.training_inputs.columns if column.startswith("input_")]
    )
    expected_target_cols = len(
        [column for column in dataset.training_targets.columns if column.startswith("rate_const_")]
    )
    return [
        {
            "check_name": "parsed_case_count",
            "status": "pass" if input_rows == expected_total_cases else "fail",
            "expected": expected_total_cases,
            "observed": input_rows,
            "details": "Parsed training_inputs row count",
        },
        {
            "check_name": "parsed_input_columns",
            "status": "pass" if expected_input_cols == expected_input_species else "fail",
            "expected": expected_input_species,
            "observed": expected_input_cols,
            "details": "Number of input feature columns with prefix input_",
        },
        {
            "check_name": "parsed_input_table_rows_match_target_rows",
            "status": "pass" if input_rows == target_rows else "fail",
            "expected": input_rows,
            "observed": target_rows,
            "details": "training_inputs and training_targets should have the same row count",
        },
        {
            "check_name": "parsed_input_table_column_count",
            "status": "pass",
            "expected": input_cols,
            "observed": input_cols,
            "details": "Full parsed training_inputs column count including metadata",
        },
        {
            "check_name": "parsed_target_columns",
            "status": "pass" if expected_target_cols == expected_rate_count else "fail",
            "expected": expected_rate_count,
            "observed": expected_target_cols,
            "details": "Number of target columns with prefix rate_const_",
        },
        {
            "check_name": "parsed_target_table_column_count",
            "status": "pass",
            "expected": target_cols,
            "observed": target_cols,
            "details": "Full parsed training_targets column count including metadata",
        },
        {
            "check_name": "density_group_count",
            "status": "pass" if density_groups == expected_density_groups else "fail",
            "expected": expected_density_groups,
            "observed": density_groups,
            "details": "Unique density groups in parsed inputs",
        },
        {
            "check_name": "e_over_n_count",
            "status": "pass" if e_over_n_values == expected_local_case_count else "fail",
            "expected": expected_local_case_count,
            "observed": e_over_n_values,
            "details": "Unique E/N values in parsed inputs",
        },
    ]
