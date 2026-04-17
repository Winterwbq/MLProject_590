from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.analysis_utils import ensure_dir, normalize_v03_analysis_path, numeric_summary, save_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run power-aware RATE CONST analysis on parsed NERS590 CSV tables."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ners590_v03_analysis" / "parsed",
        help="Directory containing parsed NERS590 CSV tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ners590_v03_analysis" / "analysis",
        help="Directory for analysis CSV outputs.",
    )
    args = parser.parse_args()
    args.parsed_dir = normalize_v03_analysis_path(args.parsed_dir, "parsed", log_prefix="[rate-analysis]")
    args.output_dir = normalize_v03_analysis_path(args.output_dir, "analysis", log_prefix="[rate-analysis]")

    ensure_dir(args.output_dir)

    parser_summary = pd.read_csv(args.parsed_dir / "parser_summary.csv")
    parser_file_summary = pd.read_csv(args.parsed_dir / "parser_file_summary.csv")
    case_features = pd.read_csv(args.parsed_dir / "case_features.csv")
    training_inputs = pd.read_csv(args.parsed_dir / "training_inputs.csv")
    training_targets = pd.read_csv(args.parsed_dir / "training_targets.csv")
    species_map = pd.read_csv(args.parsed_dir / "species_map.csv")
    reaction_map = pd.read_csv(args.parsed_dir / "reaction_map.csv")

    input_columns = [column for column in training_inputs.columns if column.startswith("input_")]
    target_columns = [column for column in training_targets.columns if column.startswith("rate_const_")]
    metadata_columns = [
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

    dataset_summary = pd.DataFrame(
        [
            {"metric": "total_cases", "value": len(training_inputs)},
            {"metric": "power_level_count", "value": training_inputs["power_label"].nunique()},
            {"metric": "power_values_mj_count", "value": training_inputs["power_mj"].nunique()},
            {"metric": "density_group_count_total", "value": training_inputs["density_group_id"].nunique()},
            {
                "metric": "density_group_count_per_power",
                "value": training_inputs.groupby("power_label")["density_group_in_file_id"].nunique().median(),
            },
            {"metric": "local_case_count", "value": training_inputs["local_case_id"].nunique()},
            {"metric": "unique_e_over_n_count", "value": training_inputs["e_over_n_v_cm2"].nunique()},
            {"metric": "input_species_count", "value": len(input_columns)},
            {"metric": "target_rate_count", "value": len(target_columns)},
            {"metric": "input_value_entry_count", "value": len(training_inputs) * len(input_columns)},
            {"metric": "target_value_entry_count", "value": len(training_targets) * len(target_columns)},
        ]
    )
    save_csv(dataset_summary, args.output_dir / "dataset_summary.csv")
    save_csv(parser_summary, args.output_dir / "parser_summary_copy.csv")
    save_csv(parser_file_summary, args.output_dir / "parser_file_summary_copy.csv")

    power_summary = (
        training_inputs.groupby(["power_label", "power_mj"], as_index=False)
        .agg(
            case_count=("global_case_id", "size"),
            density_group_count=("density_group_in_file_id", "nunique"),
            local_case_count=("local_case_id", "nunique"),
            unique_e_over_n_count=("e_over_n_v_cm2", "nunique"),
            min_e_over_n=("e_over_n_v_cm2", "min"),
            max_e_over_n=("e_over_n_v_cm2", "max"),
        )
        .sort_values("power_mj")
    )
    save_csv(power_summary, args.output_dir / "power_summary.csv")

    e_over_n_by_power = (
        training_inputs.groupby(["power_label", "power_mj", "local_case_id", "e_over_n_v_cm2"], as_index=False)
        .agg(case_count=("global_case_id", "size"))
        .sort_values(["power_mj", "local_case_id"])
    )
    save_csv(e_over_n_by_power, args.output_dir / "e_over_n_by_power.csv")

    density_group_summary = (
        training_inputs.groupby(["power_label", "power_mj", "density_group_in_file_id"], as_index=False)
        .agg(
            case_count=("global_case_id", "size"),
            local_case_min=("local_case_id", "min"),
            local_case_max=("local_case_id", "max"),
            unique_e_over_n_count=("e_over_n_v_cm2", "nunique"),
        )
        .sort_values(["power_mj", "density_group_in_file_id"])
    )
    save_csv(density_group_summary, args.output_dir / "density_group_summary.csv")

    input_matrix = training_inputs[input_columns].to_numpy(dtype=float)
    input_species_summary_rows = []
    for row in species_map.itertuples(index=False):
        values = training_inputs[row.species_column].to_numpy(dtype=float)
        input_species_summary_rows.append(
            {
                "species_id": row.species_id,
                "species_label": row.species_label,
                "input_column": row.species_column,
                **numeric_summary(values),
                "nonzero_frequency": float(np.mean(values != 0.0)),
            }
        )
    input_species_summary = pd.DataFrame(input_species_summary_rows).sort_values(
        ["nonzero_frequency", "mean"], ascending=[False, False]
    )
    save_csv(input_species_summary, args.output_dir / "input_species_summary.csv")

    input_species_by_power_rows = []
    for (power_label, power_mj), group in training_inputs.groupby(["power_label", "power_mj"], sort=False):
        for row in species_map.itertuples(index=False):
            values = group[row.species_column].to_numpy(dtype=float)
            input_species_by_power_rows.append(
                {
                    "power_label": power_label,
                    "power_mj": power_mj,
                    "species_id": row.species_id,
                    "species_label": row.species_label,
                    "input_column": row.species_column,
                    **numeric_summary(values),
                    "nonzero_frequency": float(np.mean(values != 0.0)),
                }
            )
    input_species_by_power = pd.DataFrame(input_species_by_power_rows)
    save_csv(input_species_by_power, args.output_dir / "input_species_by_power_summary.csv")

    group_variation_rows = []
    for (power_label, density_group_in_file_id), group in training_inputs.groupby(
        ["power_label", "density_group_in_file_id"], sort=False
    ):
        variation = group[input_columns].max(axis=0) - group[input_columns].min(axis=0)
        group_variation_rows.append(
            {
                "power_label": power_label,
                "density_group_in_file_id": density_group_in_file_id,
                "case_count": int(len(group)),
                "varied_species_count_within_group": int((variation != 0.0).sum()),
                "max_species_range_within_group": float(variation.max()),
            }
        )
    save_csv(pd.DataFrame(group_variation_rows), args.output_dir / "input_group_stability.csv")

    aligned_group_rows = []
    for density_group_in_file_id, group in training_inputs.groupby("density_group_in_file_id"):
        variation = group.groupby("power_label")[input_columns].first()
        span = variation.max(axis=0) - variation.min(axis=0)
        aligned_group_rows.append(
            {
                "density_group_in_file_id": density_group_in_file_id,
                "available_power_levels": int(variation.shape[0]),
                "species_changed_across_powers": int((span != 0.0).sum()),
                "max_species_difference_across_powers": float(span.max()),
                "mean_species_difference_across_powers": float(span.mean()),
            }
        )
    save_csv(pd.DataFrame(aligned_group_rows), args.output_dir / "composition_alignment_across_powers.csv")

    case_summary = case_features[
        metadata_columns
        + [
            "e_over_n_v_cm2",
            "average_electron_energy_ev",
            "updated_electron_density_per_cc",
            "drift_velocity_cm_per_s",
            "mobility_cm2_per_v_s",
            "ionization_coefficient_per_cm",
            "total_power_loss_ev_cm3_per_s",
            "input_mole_fraction_sum",
            "input_nonzero_species_count",
        ]
    ].copy()
    save_csv(case_summary, args.output_dir / "case_summary.csv")

    case_feature_by_power = (
        case_summary.groupby(["power_label", "power_mj"], as_index=False)
        .agg(
            average_electron_energy_mean=("average_electron_energy_ev", "mean"),
            average_electron_energy_median=("average_electron_energy_ev", "median"),
            updated_electron_density_mean=("updated_electron_density_per_cc", "mean"),
            drift_velocity_mean=("drift_velocity_cm_per_s", "mean"),
            mobility_mean=("mobility_cm2_per_v_s", "mean"),
            ionization_coefficient_mean=("ionization_coefficient_per_cm", "mean"),
            total_power_loss_mean=("total_power_loss_ev_cm3_per_s", "mean"),
            input_nonzero_species_count_mean=("input_nonzero_species_count", "mean"),
        )
        .sort_values("power_mj")
    )
    save_csv(case_feature_by_power, args.output_dir / "case_features_by_power.csv")

    target_matrix = training_targets[target_columns].to_numpy(dtype=float)
    rate_overall_summary = pd.DataFrame([numeric_summary(target_matrix.ravel())])
    save_csv(rate_overall_summary, args.output_dir / "rate_overall_summary.csv")

    rate_by_power_rows = []
    for (power_label, power_mj), group_idx in training_targets.groupby(["power_label", "power_mj"]).groups.items():
        values = training_targets.loc[group_idx, target_columns].to_numpy(dtype=float).ravel()
        rate_by_power_rows.append(
            {"power_label": power_label, "power_mj": power_mj, **numeric_summary(values)}
        )
    save_csv(pd.DataFrame(rate_by_power_rows).sort_values("power_mj"), args.output_dir / "rate_by_power_summary.csv")

    case_rate_rows = []
    for row in training_targets[metadata_columns + target_columns].itertuples(index=False):
        values = np.array(row[len(metadata_columns):], dtype=float)
        case_rate_rows.append(
            {
                "global_case_id": row.global_case_id,
                "power_label": row.power_label,
                "power_mj": row.power_mj,
                "density_group_id": row.density_group_id,
                "density_group_in_file_id": row.density_group_in_file_id,
                "local_case_id": row.local_case_id,
                "nonzero_rate_count": int(np.sum(values != 0.0)),
                "max_rate_const": float(np.max(values)),
                "sum_rate_const": float(np.sum(values)),
                "median_rate_const": float(np.median(values)),
            }
        )
    case_rate_summary = pd.DataFrame(case_rate_rows)
    save_csv(case_rate_summary, args.output_dir / "rate_case_summary.csv")

    reaction_summary_rows = []
    for reaction in reaction_map.itertuples(index=False):
        values = training_targets[reaction.rate_const_column].to_numpy(dtype=float)
        reaction_summary_rows.append(
            {
                "reaction_id": reaction.reaction_id,
                "reaction_label": reaction.reaction_label,
                "target_column": reaction.rate_const_column,
                **numeric_summary(values),
                "nonzero_frequency": float(np.mean(values != 0.0)),
            }
        )
    reaction_summary = pd.DataFrame(reaction_summary_rows).sort_values(
        ["mean", "nonzero_frequency"], ascending=[False, False]
    )
    save_csv(reaction_summary, args.output_dir / "reaction_summary.csv")

    reaction_by_power_rows = []
    for (power_label, power_mj), group in training_targets.groupby(["power_label", "power_mj"], sort=False):
        for reaction in reaction_map.itertuples(index=False):
            values = group[reaction.rate_const_column].to_numpy(dtype=float)
            reaction_by_power_rows.append(
                {
                    "power_label": power_label,
                    "power_mj": power_mj,
                    "reaction_id": reaction.reaction_id,
                    "reaction_label": reaction.reaction_label,
                    "target_column": reaction.rate_const_column,
                    **numeric_summary(values),
                    "nonzero_frequency": float(np.mean(values != 0.0)),
                }
            )
    reaction_by_power = pd.DataFrame(reaction_by_power_rows)
    save_csv(reaction_by_power, args.output_dir / "reaction_by_power_summary.csv")

    power_sensitivity_rows = []
    base_target_frame = training_targets[
        metadata_columns + target_columns
    ].sort_values(["density_group_in_file_id", "local_case_id", "power_mj"])
    for reaction in reaction_map.itertuples(index=False):
        pivot = base_target_frame.pivot_table(
            index=["density_group_in_file_id", "local_case_id"],
            columns="power_label",
            values=reaction.rate_const_column,
            aggfunc="first",
        )
        if pivot.empty:
            continue
        filled = pivot.fillna(0.0).to_numpy(dtype=float)
        log_filled = np.log10(filled + max(np.min(filled[filled > 0.0]) / 10.0 if np.any(filled > 0.0) else 1e-30, 1e-30))
        power_sensitivity_rows.append(
            {
                "reaction_id": reaction.reaction_id,
                "reaction_label": reaction.reaction_label,
                "target_column": reaction.rate_const_column,
                "matched_density_local_pairs": int(pivot.shape[0]),
                "available_power_levels": int(pivot.shape[1]),
                "mean_std_across_powers": float(np.mean(np.std(filled, axis=1))),
                "mean_log_std_across_powers": float(np.mean(np.std(log_filled, axis=1))),
                "max_value_across_pairs": float(np.max(filled)),
            }
        )
    power_sensitivity = pd.DataFrame(power_sensitivity_rows).sort_values(
        "mean_log_std_across_powers", ascending=False
    )
    save_csv(power_sensitivity, args.output_dir / "reaction_power_sensitivity_summary.csv")

    power_grid_consistency = (
        training_inputs.groupby(["power_label", "local_case_id"], as_index=False)
        .agg(
            case_count=("global_case_id", "size"),
            e_over_n_unique=("e_over_n_v_cm2", "nunique"),
            e_over_n_value=("e_over_n_v_cm2", "first"),
        )
        .sort_values(["power_label", "local_case_id"])
    )
    save_csv(power_grid_consistency, args.output_dir / "power_grid_consistency.csv")

    manifest = pd.DataFrame(
        [
            {"path": str(args.output_dir / "dataset_summary.csv"), "description": "Top-level merged dataset counts and dimensions."},
            {"path": str(args.output_dir / "power_summary.csv"), "description": "Case and grid counts per power level."},
            {"path": str(args.output_dir / "e_over_n_by_power.csv"), "description": "E/N grid reused across power levels."},
            {"path": str(args.output_dir / "input_species_summary.csv"), "description": "Overall input-species statistics."},
            {"path": str(args.output_dir / "input_species_by_power_summary.csv"), "description": "Input-species statistics by power."},
            {"path": str(args.output_dir / "composition_alignment_across_powers.csv"), "description": "How matched density groups differ across powers."},
            {"path": str(args.output_dir / "case_features_by_power.csv"), "description": "Case-feature averages by power."},
            {"path": str(args.output_dir / "rate_overall_summary.csv"), "description": "Overall rate-constant distribution."},
            {"path": str(args.output_dir / "rate_by_power_summary.csv"), "description": "Rate-constant distribution by power."},
            {"path": str(args.output_dir / "reaction_summary.csv"), "description": "Per-reaction overall statistics."},
            {"path": str(args.output_dir / "reaction_by_power_summary.csv"), "description": "Per-reaction statistics by power."},
            {"path": str(args.output_dir / "reaction_power_sensitivity_summary.csv"), "description": "Reaction sensitivity to power across matched group/local coordinates."},
        ]
    )
    save_csv(manifest, args.output_dir / "analysis_manifest.csv")
    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
