from __future__ import annotations

import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CASE_BLOCK_LINES = 344
EXPECTED_LOCAL_CASE_COUNT = 29
EXPECTED_DENSITY_GROUP_COUNT = 21
EXPECTED_INPUT_SPECIES_COUNT = 49
EXPECTED_RATE_COUNT = 204
EXPECTED_POWER_SPECIES_COUNT = 49

CASE_HEADER_RE = re.compile(r"CASE=\s*(\d+)")
INPUT_ROW_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$")
RATE_ROW_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(.*?)\s*$")
POWER_ROW_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s*$")
NUMERIC_TOKEN_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+|[+-]\d+)?"
)


@dataclass
class ParsedCase:
    global_case_id: int
    density_group_id: int
    local_case_id: int
    gas_ion_temperature_k: float
    electron_fractional_ionization: float
    initial_electron_temperature_ev: float
    maximum_iterations_for_edf: int
    e_over_n_v_cm2: float
    average_electron_energy_ev: float
    equivalent_electron_temperature_ev: float
    convergence_for_eedf_raw: str
    convergence_for_eedf: float
    updated_electron_density_per_cc: float
    updated_fractional_ionization: float
    drift_velocity_cm_per_s: float
    mobility_cm2_per_v_s: float
    diffusion_coefficient_cm2_per_s: float
    ionization_coefficient_per_cm: float
    mom_transfer_collision_frequency_per_s: float
    maximum_time_for_eedf_integration_s: float
    number_of_updates_for_eedf: int
    collisional_power_deposition_ev_cm3_per_s: float
    je_over_collision_power: float
    total_power_loss_ev_cm3_per_s: float
    input_species: list[tuple[int, str, float]]
    rate_rows: list[tuple[int, str, float, float, float]]
    power_species: list[tuple[int, str, float]]


def normalize_numeric_token(token: str) -> str:
    cleaned = token.strip().rstrip(")")
    if "E" not in cleaned.upper():
        malformed = re.fullmatch(r"([+-]?\d+(?:\.\d*)?)([+-]\d+)", cleaned)
        if malformed:
            cleaned = f"{malformed.group(1)}E{malformed.group(2)}"
    return cleaned


def parse_float_token(token: str) -> float:
    return float(normalize_numeric_token(token))


def extract_first_numeric_token(text: str) -> str:
    match = NUMERIC_TOKEN_RE.search(text)
    if not match:
        raise ValueError(f"Could not find a numeric token in line: {text!r}")
    return match.group(0)


def extract_first_float(text: str) -> float:
    return parse_float_token(extract_first_numeric_token(text))


def extract_first_int(text: str) -> int:
    return int(float(extract_first_numeric_token(text)))


def find_line_index(block: list[str], needle: str) -> int:
    for idx, line in enumerate(block):
        if line.strip() == needle:
            return idx
    raise ValueError(f"Could not find section marker {needle!r}")


def find_line_startswith(block: list[str], needle: str) -> int:
    for idx, line in enumerate(block):
        if line.strip().startswith(needle):
            return idx
    raise ValueError(f"Could not find section starting with {needle!r}")


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_").lower()
    return sanitized or "value"


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def compute_numeric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    stats = {
        "count": float(len(values)),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": percentile(values, 0.50),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "sum": math.fsum(values),
        "zero_count": float(sum(1 for value in values if value == 0.0)),
        "nonzero_count": float(sum(1 for value in values if value != 0.0)),
        "positive_count": float(sum(1 for value in values if value > 0.0)),
        "negative_count": float(sum(1 for value in values if value < 0.0)),
        "unique_count": float(len(set(values))),
    }
    return stats


def build_log10_histogram(values: list[float]) -> list[dict[str, object]]:
    positive_values = [value for value in values if value > 0.0]
    if not positive_values:
        return []
    counts = Counter(math.floor(math.log10(value)) for value in positive_values)
    rows = []
    for exponent in sorted(counts):
        rows.append(
            {
                "bin_floor_log10": exponent,
                "bin_start": f"1e{exponent}",
                "bin_end_exclusive": f"1e{exponent + 1}",
                "count": counts[exponent],
            }
        )
    return rows


def block_to_case(
    block: list[str],
    global_case_id: int,
    density_group_id: int,
) -> ParsedCase:
    header_match = CASE_HEADER_RE.search(block[0])
    if not header_match:
        raise ValueError(f"Could not parse case header: {block[0]!r}")
    local_case_id = int(header_match.group(1))

    input_start = find_line_index(block, "INPUT GAS MOLE FRACTIONS:")
    final_values_start = find_line_index(block, "FINAL VALUES")
    transport_start = find_line_index(block, "TRANSPORT COEFFICIENTS")
    rate_start = find_line_index(block, "RATE CONST  SUPER RATE (CC/S)  FRAC POWER")
    total_power_idx = find_line_startswith(
        block, "TOTAL POWER LOSS BY ELECTRONS (eV-cm3/s)="
    )
    power_start = find_line_index(block, "FRACTIONAL POWER DEPOSITION BY SPECIES")

    input_species: list[tuple[int, str, float]] = []
    for line in block[input_start + 1 : final_values_start]:
        match = INPUT_ROW_RE.match(line)
        if match:
            species_id = int(match.group(1))
            label = match.group(2)
            value = parse_float_token(match.group(3))
            input_species.append((species_id, label, value))

    rate_rows: list[tuple[int, str, float, float, float]] = []
    for line in block[rate_start + 1 : total_power_idx]:
        match = RATE_ROW_RE.match(line)
        if match:
            rate_const = parse_float_token(match.group(1))
            super_rate = parse_float_token(match.group(2))
            frac_power = parse_float_token(match.group(3))
            reaction_label = match.group(4).strip()
            rate_rows.append(
                (
                    len(rate_rows) + 1,
                    reaction_label,
                    rate_const,
                    super_rate,
                    frac_power,
                )
            )

    power_species: list[tuple[int, str, float]] = []
    for line in block[power_start + 1 :]:
        match = POWER_ROW_RE.match(line)
        if match:
            label = match.group(1)
            value = parse_float_token(match.group(2))
            power_species.append((len(power_species) + 1, label, value))

    scalar_lines = {line.strip(): line.strip() for line in block}
    gas_ion_temperature_k = extract_first_float(
        next(line for line in scalar_lines if line.startswith("GAS/ION TEMPERATURE="))
    )
    electron_fractional_ionization = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("ELECTRON FRACTIONAL IONIZATION=")
        )
    )
    initial_electron_temperature_ev = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("INITIAL ELECTRON TEMPERATURE=")
        )
    )
    maximum_iterations_for_edf = extract_first_int(
        next(
            line
            for line in scalar_lines
            if line.startswith("MAXIMUM ITERATIONS FOR EDF=")
        )
    )
    e_over_n_v_cm2 = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("ELECTRIC FIELD/NUMBER DENSITY=")
        )
    )
    average_electron_energy_ev = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("AVERAGE ELECTRON ENERGY=")
        )
    )
    equivalent_electron_temperature_ev = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("EQUIVALENT ELECTRON TEMPERATURE=")
        )
    )
    convergence_line = next(
        line for line in scalar_lines if line.startswith("CONVERGANCE FOR EEDF=")
    )
    convergence_for_eedf_raw = convergence_line.split("=", 1)[1].strip()
    convergence_for_eedf = extract_first_float(convergence_for_eedf_raw)
    updated_electron_density_per_cc = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("UPDATED ELECTRON DENSITY=")
        )
    )
    updated_fractional_ionization = extract_first_float(
        next(line for line in scalar_lines if line.startswith("(FRACTIONAL IONIZATION="))
    )
    drift_velocity_cm_per_s = extract_first_float(
        next(line for line in scalar_lines if line.startswith("DRIFT VELOCITY="))
    )
    mobility_cm2_per_v_s = extract_first_float(
        next(line for line in scalar_lines if line.startswith("MOBILITY="))
    )
    diffusion_coefficient_cm2_per_s = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("DIFFUSION COEFFICIENT=")
        )
    )
    ionization_coefficient_per_cm = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("IONIZATION COEFFICIENT=")
        )
    )
    mom_transfer_collision_frequency_per_s = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("MOM TRANSFER COLLISION FREQUENCY=")
        )
    )
    maximum_time_for_eedf_integration_s = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("MAXIMUM TIME FOR EEDF INTEGRATION=")
        )
    )
    number_of_updates_for_eedf = extract_first_int(
        next(
            line
            for line in scalar_lines
            if line.startswith("NUMBER OF UPDATES FOR EEDF=")
        )
    )
    collisional_power_deposition_ev_cm3_per_s = extract_first_float(
        next(
            line
            for line in scalar_lines
            if line.startswith("COLLISIONAL POWER DEPOSIITON")
        )
    )
    je_over_collision_power = extract_first_float(
        next(line for line in scalar_lines if line.startswith("JE/COLLISION POWER ="))
    )
    total_power_loss_line = block[total_power_idx]
    total_power_loss_ev_cm3_per_s = extract_first_float(
        total_power_loss_line.split("=", 1)[1]
    )

    return ParsedCase(
        global_case_id=global_case_id,
        density_group_id=density_group_id,
        local_case_id=local_case_id,
        gas_ion_temperature_k=gas_ion_temperature_k,
        electron_fractional_ionization=electron_fractional_ionization,
        initial_electron_temperature_ev=initial_electron_temperature_ev,
        maximum_iterations_for_edf=maximum_iterations_for_edf,
        e_over_n_v_cm2=e_over_n_v_cm2,
        average_electron_energy_ev=average_electron_energy_ev,
        equivalent_electron_temperature_ev=equivalent_electron_temperature_ev,
        convergence_for_eedf_raw=convergence_for_eedf_raw,
        convergence_for_eedf=convergence_for_eedf,
        updated_electron_density_per_cc=updated_electron_density_per_cc,
        updated_fractional_ionization=updated_fractional_ionization,
        drift_velocity_cm_per_s=drift_velocity_cm_per_s,
        mobility_cm2_per_v_s=mobility_cm2_per_v_s,
        diffusion_coefficient_cm2_per_s=diffusion_coefficient_cm2_per_s,
        ionization_coefficient_per_cm=ionization_coefficient_per_cm,
        mom_transfer_collision_frequency_per_s=mom_transfer_collision_frequency_per_s,
        maximum_time_for_eedf_integration_s=maximum_time_for_eedf_integration_s,
        number_of_updates_for_eedf=number_of_updates_for_eedf,
        collisional_power_deposition_ev_cm3_per_s=collisional_power_deposition_ev_cm3_per_s,
        je_over_collision_power=je_over_collision_power,
        total_power_loss_ev_cm3_per_s=total_power_loss_ev_cm3_per_s,
        input_species=input_species,
        rate_rows=rate_rows,
        power_species=power_species,
    )


def parse_dataset(raw_path: Path) -> tuple[list[ParsedCase], dict[str, object]]:
    lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_indices = [
        index for index, line in enumerate(lines) if CASE_HEADER_RE.search(line)
    ]
    if not header_indices:
        raise ValueError(f"No case headers found in {raw_path}")

    header_indices.append(len(lines))
    cases: list[ParsedCase] = []
    expected_input_labels: list[str] | None = None
    expected_reaction_labels: list[str] | None = None
    expected_power_labels: list[str] | None = None

    density_group_id = 0
    previous_local_case_id = None

    for case_index, start_idx in enumerate(header_indices[:-1], start=1):
        end_idx = header_indices[case_index]
        block = lines[start_idx:end_idx]
        minimum_expected_lines = CASE_BLOCK_LINES if case_index < len(header_indices) - 1 else CASE_BLOCK_LINES - 2
        if len(block) < minimum_expected_lines or len(block) > CASE_BLOCK_LINES:
            raise ValueError(
                f"Case {case_index} has {len(block)} lines instead of {CASE_BLOCK_LINES}"
            )

        header_match = CASE_HEADER_RE.search(block[0])
        if not header_match:
            raise ValueError(f"Could not parse header for case {case_index}")
        local_case_id = int(header_match.group(1))

        if previous_local_case_id is None or local_case_id == 1:
            density_group_id += 1
        previous_local_case_id = local_case_id

        parsed_case = block_to_case(
            block=block,
            global_case_id=case_index,
            density_group_id=density_group_id,
        )

        input_labels = [label for _, label, _ in parsed_case.input_species]
        reaction_labels = [label for _, label, _, _, _ in parsed_case.rate_rows]
        power_labels = [label for _, label, _ in parsed_case.power_species]

        if len(parsed_case.input_species) != EXPECTED_INPUT_SPECIES_COUNT:
            raise ValueError(
                f"Case {case_index} has {len(parsed_case.input_species)} input species rows"
            )
        if len(parsed_case.rate_rows) != EXPECTED_RATE_COUNT:
            raise ValueError(
                f"Case {case_index} has {len(parsed_case.rate_rows)} rate rows"
            )
        if len(parsed_case.power_species) != EXPECTED_POWER_SPECIES_COUNT:
            raise ValueError(
                f"Case {case_index} has {len(parsed_case.power_species)} power rows"
            )

        if expected_input_labels is None:
            expected_input_labels = input_labels
        elif input_labels != expected_input_labels:
            raise ValueError(f"Input species label mismatch in case {case_index}")

        if expected_reaction_labels is None:
            expected_reaction_labels = reaction_labels
        elif reaction_labels != expected_reaction_labels:
            raise ValueError(f"Reaction label mismatch in case {case_index}")

        if expected_power_labels is None:
            expected_power_labels = power_labels
        elif power_labels != expected_power_labels:
            raise ValueError(f"Power species label mismatch in case {case_index}")

        cases.append(parsed_case)

    metadata = {
        "source_path": str(raw_path),
        "total_cases": len(cases),
        "density_group_count": density_group_id,
        "expected_local_case_count": EXPECTED_LOCAL_CASE_COUNT,
        "input_species_labels": expected_input_labels or [],
        "reaction_labels": expected_reaction_labels or [],
        "power_species_labels": expected_power_labels or [],
        "unique_local_case_ids": sorted({case.local_case_id for case in cases}),
        "unique_e_over_n_values": sorted({case.e_over_n_v_cm2 for case in cases}),
        "case_block_lines": CASE_BLOCK_LINES,
    }
    return cases, metadata


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_case_feature_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        rows.append(
            {
                "global_case_id": case.global_case_id,
                "density_group_id": case.density_group_id,
                "local_case_id": case.local_case_id,
                "gas_ion_temperature_k": case.gas_ion_temperature_k,
                "electron_fractional_ionization": case.electron_fractional_ionization,
                "initial_electron_temperature_ev": case.initial_electron_temperature_ev,
                "maximum_iterations_for_edf": case.maximum_iterations_for_edf,
                "e_over_n_v_cm2": case.e_over_n_v_cm2,
                "average_electron_energy_ev": case.average_electron_energy_ev,
                "equivalent_electron_temperature_ev": case.equivalent_electron_temperature_ev,
                "convergence_for_eedf_raw": case.convergence_for_eedf_raw,
                "convergence_for_eedf": case.convergence_for_eedf,
                "updated_electron_density_per_cc": case.updated_electron_density_per_cc,
                "updated_fractional_ionization": case.updated_fractional_ionization,
                "drift_velocity_cm_per_s": case.drift_velocity_cm_per_s,
                "mobility_cm2_per_v_s": case.mobility_cm2_per_v_s,
                "diffusion_coefficient_cm2_per_s": case.diffusion_coefficient_cm2_per_s,
                "ionization_coefficient_per_cm": case.ionization_coefficient_per_cm,
                "mom_transfer_collision_frequency_per_s": case.mom_transfer_collision_frequency_per_s,
                "maximum_time_for_eedf_integration_s": case.maximum_time_for_eedf_integration_s,
                "number_of_updates_for_eedf": case.number_of_updates_for_eedf,
                "collisional_power_deposition_ev_cm3_per_s": case.collisional_power_deposition_ev_cm3_per_s,
                "je_over_collision_power": case.je_over_collision_power,
                "total_power_loss_ev_cm3_per_s": case.total_power_loss_ev_cm3_per_s,
                "input_mole_fraction_sum": math.fsum(
                    value for _, _, value in case.input_species
                ),
                "input_nonzero_species_count": sum(
                    1 for _, _, value in case.input_species if value != 0.0
                ),
            }
        )
    return rows


def build_species_map_rows(species_labels: list[str]) -> list[dict[str, object]]:
    rows = []
    for species_id, label in enumerate(species_labels, start=1):
        rows.append(
            {
                "species_id": species_id,
                "species_label": label,
                "species_column": f"input_{species_id:02d}_{sanitize_label(label)}",
            }
        )
    return rows


def build_reaction_map_rows(reaction_labels: list[str]) -> list[dict[str, object]]:
    rows = []
    for reaction_id, label in enumerate(reaction_labels, start=1):
        rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_label": label,
                "rate_const_column": f"rate_const_{reaction_id:03d}",
                "super_rate_column": f"super_rate_{reaction_id:03d}",
                "frac_power_column": f"frac_power_{reaction_id:03d}",
            }
        )
    return rows


def build_input_long_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        for species_id, label, value in case.input_species:
            rows.append(
                {
                    "global_case_id": case.global_case_id,
                    "density_group_id": case.density_group_id,
                    "local_case_id": case.local_case_id,
                    "species_id": species_id,
                    "species_label": label,
                    "mole_fraction": value,
                }
            )
    return rows


def build_power_long_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        for species_id, label, value in case.power_species:
            rows.append(
                {
                    "global_case_id": case.global_case_id,
                    "density_group_id": case.density_group_id,
                    "local_case_id": case.local_case_id,
                    "species_id": species_id,
                    "species_label": label,
                    "fractional_power_deposition": value,
                }
            )
    return rows


def build_rate_long_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        for reaction_id, label, rate_const, super_rate, frac_power in case.rate_rows:
            rows.append(
                {
                    "global_case_id": case.global_case_id,
                    "density_group_id": case.density_group_id,
                    "local_case_id": case.local_case_id,
                    "reaction_id": reaction_id,
                    "reaction_label": label,
                    "rate_const": rate_const,
                    "super_rate_cc_per_s": super_rate,
                    "frac_power": frac_power,
                }
            )
    return rows


def build_training_input_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        row: dict[str, object] = {
            "global_case_id": case.global_case_id,
            "density_group_id": case.density_group_id,
            "local_case_id": case.local_case_id,
            "e_over_n_v_cm2": case.e_over_n_v_cm2,
        }
        for species_id, label, value in case.input_species:
            row[f"input_{species_id:02d}_{sanitize_label(label)}"] = value
        rows.append(row)
    return rows


def build_training_target_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        row: dict[str, object] = {
            "global_case_id": case.global_case_id,
            "density_group_id": case.density_group_id,
            "local_case_id": case.local_case_id,
        }
        for reaction_id, _label, rate_const, _super_rate, _frac_power in case.rate_rows:
            row[f"rate_const_{reaction_id:03d}"] = rate_const
        rows.append(row)
    return rows


def build_parser_summary_rows(
    cases: list[ParsedCase], metadata: dict[str, object]
) -> list[dict[str, object]]:
    group_counts = Counter(case.density_group_id for case in cases)
    group_sizes = list(group_counts.values())
    summary_rows = [
        {"metric": "source_path", "value": metadata["source_path"]},
        {"metric": "total_cases", "value": len(cases)},
        {"metric": "density_group_count", "value": metadata["density_group_count"]},
        {"metric": "expected_density_group_count", "value": EXPECTED_DENSITY_GROUP_COUNT},
        {"metric": "local_case_count_expected", "value": EXPECTED_LOCAL_CASE_COUNT},
        {
            "metric": "local_case_count_observed",
            "value": len(metadata["unique_local_case_ids"]),
        },
        {
            "metric": "unique_e_over_n_values",
            "value": len(metadata["unique_e_over_n_values"]),
        },
        {"metric": "case_block_lines", "value": metadata["case_block_lines"]},
        {"metric": "input_species_count", "value": EXPECTED_INPUT_SPECIES_COUNT},
        {"metric": "rate_count", "value": EXPECTED_RATE_COUNT},
        {"metric": "power_species_count", "value": EXPECTED_POWER_SPECIES_COUNT},
        {"metric": "min_cases_per_density_group", "value": min(group_sizes)},
        {"metric": "max_cases_per_density_group", "value": max(group_sizes)},
    ]
    return summary_rows


def write_parser_outputs(
    cases: list[ParsedCase], metadata: dict[str, object], output_dir: Path
) -> dict[str, Path]:
    ensure_directory(output_dir)

    species_map_rows = build_species_map_rows(metadata["input_species_labels"])
    reaction_map_rows = build_reaction_map_rows(metadata["reaction_labels"])
    case_feature_rows = build_case_feature_rows(cases)
    input_long_rows = build_input_long_rows(cases)
    power_long_rows = build_power_long_rows(cases)
    rate_long_rows = build_rate_long_rows(cases)
    training_input_rows = build_training_input_rows(cases)
    training_target_rows = build_training_target_rows(cases)
    parser_summary_rows = build_parser_summary_rows(cases, metadata)

    output_paths = {
        "parser_summary": output_dir / "parser_summary.csv",
        "case_features": output_dir / "case_features.csv",
        "species_map": output_dir / "species_map.csv",
        "reaction_map": output_dir / "reaction_map.csv",
        "input_mole_fractions_long": output_dir / "input_mole_fractions_long.csv",
        "power_deposition_long": output_dir / "power_deposition_long.csv",
        "rate_constants_long": output_dir / "rate_constants_long.csv",
        "training_inputs": output_dir / "training_inputs.csv",
        "training_targets": output_dir / "training_targets.csv",
    }

    write_csv(output_paths["parser_summary"], parser_summary_rows, ["metric", "value"])
    write_csv(
        output_paths["case_features"],
        case_feature_rows,
        list(case_feature_rows[0].keys()),
    )
    write_csv(
        output_paths["species_map"],
        species_map_rows,
        list(species_map_rows[0].keys()),
    )
    write_csv(
        output_paths["reaction_map"],
        reaction_map_rows,
        list(reaction_map_rows[0].keys()),
    )
    write_csv(
        output_paths["input_mole_fractions_long"],
        input_long_rows,
        list(input_long_rows[0].keys()),
    )
    write_csv(
        output_paths["power_deposition_long"],
        power_long_rows,
        list(power_long_rows[0].keys()),
    )
    write_csv(
        output_paths["rate_constants_long"],
        rate_long_rows,
        list(rate_long_rows[0].keys()),
    )
    write_csv(
        output_paths["training_inputs"],
        training_input_rows,
        list(training_input_rows[0].keys()),
    )
    write_csv(
        output_paths["training_targets"],
        training_target_rows,
        list(training_target_rows[0].keys()),
    )
    return output_paths


def build_dataset_summary_rows(
    cases: list[ParsedCase], metadata: dict[str, object]
) -> list[dict[str, object]]:
    e_over_n_values = [case.e_over_n_v_cm2 for case in cases]
    all_input_values = [
        value for case in cases for _species_id, _label, value in case.input_species
    ]
    all_rate_consts = [
        rate_const
        for case in cases
        for _reaction_id, _label, rate_const, _super_rate, _frac_power in case.rate_rows
    ]
    input_sum_values = [
        math.fsum(value for _species_id, _label, value in case.input_species)
        for case in cases
    ]
    rows = [
        {"metric": "total_cases", "value": len(cases)},
        {"metric": "density_group_count", "value": metadata["density_group_count"]},
        {"metric": "expected_density_group_count", "value": EXPECTED_DENSITY_GROUP_COUNT},
        {
            "metric": "local_case_count_observed",
            "value": len(metadata["unique_local_case_ids"]),
        },
        {
            "metric": "expected_local_case_count",
            "value": EXPECTED_LOCAL_CASE_COUNT,
        },
        {
            "metric": "unique_e_over_n_values",
            "value": len(metadata["unique_e_over_n_values"]),
        },
        {"metric": "input_species_count", "value": EXPECTED_INPUT_SPECIES_COUNT},
        {"metric": "rate_count", "value": EXPECTED_RATE_COUNT},
        {
            "metric": "input_value_entries",
            "value": len(all_input_values),
        },
        {
            "metric": "rate_constant_entries",
            "value": len(all_rate_consts),
        },
        {"metric": "min_e_over_n_v_cm2", "value": min(e_over_n_values)},
        {"metric": "max_e_over_n_v_cm2", "value": max(e_over_n_values)},
        {"metric": "min_input_sum", "value": min(input_sum_values)},
        {"metric": "max_input_sum", "value": max(input_sum_values)},
        {"metric": "zero_input_value_count", "value": sum(1 for v in all_input_values if v == 0.0)},
        {"metric": "zero_rate_const_count", "value": sum(1 for v in all_rate_consts if v == 0.0)},
    ]
    return rows


def build_e_over_n_distribution_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    groups: defaultdict[float, list[ParsedCase]] = defaultdict(list)
    for case in cases:
        groups[case.e_over_n_v_cm2].append(case)

    rows = []
    for e_over_n in sorted(groups):
        grouped_cases = groups[e_over_n]
        rows.append(
            {
                "e_over_n_v_cm2": e_over_n,
                "case_count": len(grouped_cases),
                "density_group_min": min(case.density_group_id for case in grouped_cases),
                "density_group_max": max(case.density_group_id for case in grouped_cases),
                "local_case_id_min": min(case.local_case_id for case in grouped_cases),
                "local_case_id_max": max(case.local_case_id for case in grouped_cases),
            }
        )
    return rows


def build_density_group_summary_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    grouped: defaultdict[int, list[ParsedCase]] = defaultdict(list)
    for case in cases:
        grouped[case.density_group_id].append(case)

    rows = []
    for density_group_id in sorted(grouped):
        group_cases = grouped[density_group_id]
        input_ranges = []
        for species_idx in range(EXPECTED_INPUT_SPECIES_COUNT):
            values = [case.input_species[species_idx][2] for case in group_cases]
            input_ranges.append(max(values) - min(values))

        rows.append(
            {
                "density_group_id": density_group_id,
                "case_count": len(group_cases),
                "local_case_id_min": min(case.local_case_id for case in group_cases),
                "local_case_id_max": max(case.local_case_id for case in group_cases),
                "unique_e_over_n_count": len({case.e_over_n_v_cm2 for case in group_cases}),
                "min_e_over_n_v_cm2": min(case.e_over_n_v_cm2 for case in group_cases),
                "max_e_over_n_v_cm2": max(case.e_over_n_v_cm2 for case in group_cases),
                "input_sum_min": min(
                    math.fsum(value for _species_id, _label, value in case.input_species)
                    for case in group_cases
                ),
                "input_sum_max": max(
                    math.fsum(value for _species_id, _label, value in case.input_species)
                    for case in group_cases
                ),
                "varied_species_count_within_group": sum(
                    1 for species_range in input_ranges if species_range != 0.0
                ),
                "max_species_range_within_group": max(input_ranges),
            }
        )
    return rows


def build_input_case_summary_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        values = [value for _species_id, _label, value in case.input_species]
        max_species_id, max_species_label, max_species_value = max(
            case.input_species, key=lambda item: item[2]
        )
        rows.append(
            {
                "global_case_id": case.global_case_id,
                "density_group_id": case.density_group_id,
                "local_case_id": case.local_case_id,
                "input_mole_fraction_sum": math.fsum(values),
                "nonzero_species_count": sum(1 for value in values if value != 0.0),
                "zero_species_count": sum(1 for value in values if value == 0.0),
                "max_species_id": max_species_id,
                "max_species_label": max_species_label,
                "max_species_value": max_species_value,
            }
        )
    return rows


def build_input_species_summary_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for species_idx in range(EXPECTED_INPUT_SPECIES_COUNT):
        species_id = species_idx + 1
        species_label = cases[0].input_species[species_idx][1]
        values = [case.input_species[species_idx][2] for case in cases]
        stats = compute_numeric_stats(values)
        row = {"species_id": species_id, "species_label": species_label}
        row.update(stats)
        rows.append(row)
    return rows


def build_input_species_group_stability_rows(
    cases: list[ParsedCase],
) -> list[dict[str, object]]:
    grouped: defaultdict[int, list[ParsedCase]] = defaultdict(list)
    for case in cases:
        grouped[case.density_group_id].append(case)

    rows = []
    for density_group_id in sorted(grouped):
        group_cases = grouped[density_group_id]
        for species_idx in range(EXPECTED_INPUT_SPECIES_COUNT):
            species_id = species_idx + 1
            species_label = group_cases[0].input_species[species_idx][1]
            values = [case.input_species[species_idx][2] for case in group_cases]
            rows.append(
                {
                    "density_group_id": density_group_id,
                    "species_id": species_id,
                    "species_label": species_label,
                    "min_value": min(values),
                    "max_value": max(values),
                    "range_value": max(values) - min(values),
                    "is_constant_within_group": max(values) == min(values),
                }
            )
    return rows


def build_rate_case_summary_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        values = [rate_const for _reaction_id, _label, rate_const, _super_rate, _frac_power in case.rate_rows]
        max_reaction_id, max_reaction_label, max_rate_const, _super_rate, _frac_power = max(
            case.rate_rows, key=lambda item: item[2]
        )
        stats = compute_numeric_stats(values)
        row = {
            "global_case_id": case.global_case_id,
            "density_group_id": case.density_group_id,
            "local_case_id": case.local_case_id,
            "max_reaction_id": max_reaction_id,
            "max_reaction_label": max_reaction_label,
            "max_rate_const": max_rate_const,
        }
        row.update(stats)
        rows.append(row)
    return rows


def build_rate_reaction_summary_rows(cases: list[ParsedCase]) -> list[dict[str, object]]:
    rows = []
    for reaction_idx in range(EXPECTED_RATE_COUNT):
        reaction_id = reaction_idx + 1
        reaction_label = cases[0].rate_rows[reaction_idx][1]
        rate_values = [case.rate_rows[reaction_idx][2] for case in cases]
        super_rate_values = [case.rate_rows[reaction_idx][3] for case in cases]
        frac_power_values = [case.rate_rows[reaction_idx][4] for case in cases]
        row = {
            "reaction_id": reaction_id,
            "reaction_label": reaction_label,
        }
        for prefix, values in (
            ("rate_const", rate_values),
            ("super_rate", super_rate_values),
            ("frac_power", frac_power_values),
        ):
            stats = compute_numeric_stats(values)
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def build_key_value_rows(stats: dict[str, float]) -> list[dict[str, object]]:
    return [{"metric": key, "value": value} for key, value in stats.items()]


def write_analysis_outputs(
    cases: list[ParsedCase], metadata: dict[str, object], output_dir: Path
) -> dict[str, Path]:
    ensure_directory(output_dir)

    all_input_values = [
        value for case in cases for _species_id, _label, value in case.input_species
    ]
    all_rate_consts = [
        rate_const
        for case in cases
        for _reaction_id, _label, rate_const, _super_rate, _frac_power in case.rate_rows
    ]

    output_paths = {
        "dataset_summary": output_dir / "dataset_summary.csv",
        "e_over_n_distribution": output_dir / "e_over_n_distribution.csv",
        "density_group_summary": output_dir / "density_group_summary.csv",
        "input_case_summary": output_dir / "input_case_summary.csv",
        "input_species_summary": output_dir / "input_species_summary.csv",
        "input_species_group_stability": output_dir / "input_species_group_stability.csv",
        "input_value_overall_summary": output_dir / "input_value_overall_summary.csv",
        "input_value_log10_positive_histogram": output_dir
        / "input_value_log10_positive_histogram.csv",
        "rate_case_summary": output_dir / "rate_case_summary.csv",
        "rate_reaction_summary": output_dir / "rate_reaction_summary.csv",
        "rate_constant_overall_summary": output_dir / "rate_constant_overall_summary.csv",
        "rate_constant_log10_positive_histogram": output_dir
        / "rate_constant_log10_positive_histogram.csv",
    }

    dataset_summary_rows = build_dataset_summary_rows(cases, metadata)
    e_over_n_distribution_rows = build_e_over_n_distribution_rows(cases)
    density_group_summary_rows = build_density_group_summary_rows(cases)
    input_case_summary_rows = build_input_case_summary_rows(cases)
    input_species_summary_rows = build_input_species_summary_rows(cases)
    input_species_group_stability_rows = build_input_species_group_stability_rows(cases)
    input_value_overall_summary_rows = build_key_value_rows(
        compute_numeric_stats(all_input_values)
    )
    input_value_histogram_rows = build_log10_histogram(all_input_values)
    rate_case_summary_rows = build_rate_case_summary_rows(cases)
    rate_reaction_summary_rows = build_rate_reaction_summary_rows(cases)
    rate_constant_overall_summary_rows = build_key_value_rows(
        compute_numeric_stats(all_rate_consts)
    )
    rate_constant_histogram_rows = build_log10_histogram(all_rate_consts)

    write_csv(
        output_paths["dataset_summary"], dataset_summary_rows, ["metric", "value"]
    )
    write_csv(
        output_paths["e_over_n_distribution"],
        e_over_n_distribution_rows,
        list(e_over_n_distribution_rows[0].keys()),
    )
    write_csv(
        output_paths["density_group_summary"],
        density_group_summary_rows,
        list(density_group_summary_rows[0].keys()),
    )
    write_csv(
        output_paths["input_case_summary"],
        input_case_summary_rows,
        list(input_case_summary_rows[0].keys()),
    )
    write_csv(
        output_paths["input_species_summary"],
        input_species_summary_rows,
        list(input_species_summary_rows[0].keys()),
    )
    write_csv(
        output_paths["input_species_group_stability"],
        input_species_group_stability_rows,
        list(input_species_group_stability_rows[0].keys()),
    )
    write_csv(
        output_paths["input_value_overall_summary"],
        input_value_overall_summary_rows,
        ["metric", "value"],
    )
    write_csv(
        output_paths["input_value_log10_positive_histogram"],
        input_value_histogram_rows,
        ["bin_floor_log10", "bin_start", "bin_end_exclusive", "count"],
    )
    write_csv(
        output_paths["rate_case_summary"],
        rate_case_summary_rows,
        list(rate_case_summary_rows[0].keys()),
    )
    write_csv(
        output_paths["rate_reaction_summary"],
        rate_reaction_summary_rows,
        list(rate_reaction_summary_rows[0].keys()),
    )
    write_csv(
        output_paths["rate_constant_overall_summary"],
        rate_constant_overall_summary_rows,
        ["metric", "value"],
    )
    write_csv(
        output_paths["rate_constant_log10_positive_histogram"],
        rate_constant_histogram_rows,
        ["bin_floor_log10", "bin_start", "bin_end_exclusive", "count"],
    )
    return output_paths
