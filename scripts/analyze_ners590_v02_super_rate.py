from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def numeric_summary(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    summary = {
        f"{prefix}count": float(array.size),
        f"{prefix}min": float(np.min(array)),
        f"{prefix}max": float(np.max(array)),
        f"{prefix}mean": float(np.mean(array)),
        f"{prefix}median": float(np.median(array)),
        f"{prefix}std": float(np.std(array)),
        f"{prefix}p01": float(np.quantile(array, 0.01)),
        f"{prefix}p05": float(np.quantile(array, 0.05)),
        f"{prefix}p25": float(np.quantile(array, 0.25)),
        f"{prefix}p75": float(np.quantile(array, 0.75)),
        f"{prefix}p95": float(np.quantile(array, 0.95)),
        f"{prefix}p99": float(np.quantile(array, 0.99)),
        f"{prefix}zero_count": float(np.sum(array == 0.0)),
        f"{prefix}positive_count": float(np.sum(array > 0.0)),
        f"{prefix}negative_count": float(np.sum(array < 0.0)),
    }
    return summary


def safe_log_corr(x: pd.Series, y: pd.Series) -> float:
    mask = (x > 0.0) & (y > 0.0)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(np.log10(x[mask]), np.log10(y[mask]))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dedicated analysis for SUPER RATE (CC/S) on the NERS590 v02 dataset."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ners590_v02" / "parsed",
        help="Directory containing parsed v02 CSV tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ners590_v02" / "super_rate_analysis",
        help="Directory for super-rate analysis CSV outputs.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    rate_long = pd.read_csv(args.parsed_dir / "rate_constants_long.csv")
    parser_summary = pd.read_csv(args.parsed_dir / "parser_summary.csv")

    super_values = rate_long["super_rate_cc_per_s"].to_numpy(dtype=float)
    positive_super = super_values[super_values > 0.0]
    overall_row = {
        "total_entry_count": int(super_values.size),
        "positive_entry_count": int((super_values > 0.0).sum()),
        "zero_entry_count": int((super_values == 0.0).sum()),
        "positive_fraction": float(np.mean(super_values > 0.0)),
        **numeric_summary(super_values),
        "positive_only_median": float(np.median(positive_super)) if positive_super.size else float("nan"),
        "positive_only_p95": float(np.quantile(positive_super, 0.95)) if positive_super.size else float("nan"),
    }
    save_csv(pd.DataFrame([overall_row]), args.output_dir / "super_rate_overall_summary.csv")
    save_csv(parser_summary, args.output_dir / "parser_summary_copy.csv")

    by_power_rows = []
    for (power_label, power_mj), group in rate_long.groupby(["power_label", "power_mj"], sort=False):
        values = group["super_rate_cc_per_s"].to_numpy(dtype=float)
        positive_values = values[values > 0.0]
        by_power_rows.append(
            {
                "power_label": power_label,
                "power_mj": power_mj,
                "positive_fraction": float(np.mean(values > 0.0)),
                **numeric_summary(values),
                "positive_only_median": float(np.median(positive_values)) if positive_values.size else float("nan"),
                "positive_only_p95": float(np.quantile(positive_values, 0.95)) if positive_values.size else float("nan"),
            }
        )
    save_csv(pd.DataFrame(by_power_rows).sort_values("power_mj"), args.output_dir / "super_rate_by_power_summary.csv")

    case_summary = (
        rate_long.groupby(
            [
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
            ],
            as_index=False,
        )
        .agg(
            reaction_count=("reaction_id", "size"),
            nonzero_super_rate_count=("super_rate_cc_per_s", lambda s: int((s > 0.0).sum())),
            super_rate_sum=("super_rate_cc_per_s", "sum"),
            super_rate_max=("super_rate_cc_per_s", "max"),
            super_rate_median=("super_rate_cc_per_s", "median"),
            rate_const_sum=("rate_const", "sum"),
        )
    )
    save_csv(case_summary, args.output_dir / "super_rate_case_summary.csv")

    local_case_summary = (
        case_summary.groupby("local_case_id", as_index=False)
        .agg(
            case_count=("global_case_id", "size"),
            nonzero_super_rate_count_mean=("nonzero_super_rate_count", "mean"),
            super_rate_sum_mean=("super_rate_sum", "mean"),
            super_rate_max_mean=("super_rate_max", "mean"),
            super_rate_median_mean=("super_rate_median", "mean"),
        )
        .sort_values("local_case_id")
    )
    save_csv(local_case_summary, args.output_dir / "super_rate_local_case_summary.csv")

    reaction_summary_rows = []
    for (reaction_id, reaction_label), group in rate_long.groupby(["reaction_id", "reaction_label"], sort=False):
        values = group["super_rate_cc_per_s"].to_numpy(dtype=float)
        positive_values = values[values > 0.0]
        reaction_summary_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_label": reaction_label,
                "super_rate_column": f"super_rate_{int(reaction_id):03d}",
                "positive_fraction": float(np.mean(values > 0.0)),
                **numeric_summary(values),
                "positive_only_median": float(np.median(positive_values)) if positive_values.size else float("nan"),
                "positive_only_p95": float(np.quantile(positive_values, 0.95)) if positive_values.size else float("nan"),
                "log_corr_with_rate_const_positive_only": safe_log_corr(
                    group["rate_const"], group["super_rate_cc_per_s"]
                ),
            }
        )
    reaction_summary = pd.DataFrame(reaction_summary_rows).sort_values(
        ["mean", "positive_fraction"], ascending=[False, False]
    )
    save_csv(reaction_summary, args.output_dir / "super_rate_reaction_summary.csv")

    reaction_by_power_rows = []
    for (power_label, power_mj, reaction_id, reaction_label), group in rate_long.groupby(
        ["power_label", "power_mj", "reaction_id", "reaction_label"], sort=False
    ):
        values = group["super_rate_cc_per_s"].to_numpy(dtype=float)
        positive_values = values[values > 0.0]
        reaction_by_power_rows.append(
            {
                "power_label": power_label,
                "power_mj": power_mj,
                "reaction_id": reaction_id,
                "reaction_label": reaction_label,
                "positive_fraction": float(np.mean(values > 0.0)),
                **numeric_summary(values),
                "positive_only_median": float(np.median(positive_values)) if positive_values.size else float("nan"),
                "positive_only_p95": float(np.quantile(positive_values, 0.95)) if positive_values.size else float("nan"),
                "log_corr_with_rate_const_positive_only": safe_log_corr(
                    group["rate_const"], group["super_rate_cc_per_s"]
                ),
            }
        )
    save_csv(pd.DataFrame(reaction_by_power_rows), args.output_dir / "super_rate_reaction_by_power_summary.csv")

    power_sensitivity_rows = []
    for (reaction_id, reaction_label), group in rate_long.groupby(["reaction_id", "reaction_label"], sort=False):
        pivot = group.pivot_table(
            index=["density_group_in_file_id", "local_case_id"],
            columns="power_label",
            values="super_rate_cc_per_s",
            aggfunc="first",
        )
        if pivot.empty:
            continue
        filled = pivot.fillna(0.0).to_numpy(dtype=float)
        min_positive = filled[filled > 0.0].min() if np.any(filled > 0.0) else 1e-30
        eps = max(min_positive / 10.0, 1e-30)
        log_filled = np.log10(filled + eps)
        power_sensitivity_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_label": reaction_label,
                "matched_density_local_pairs": int(pivot.shape[0]),
                "available_power_levels": int(pivot.shape[1]),
                "mean_std_across_powers": float(np.mean(np.std(filled, axis=1))),
                "mean_log_std_across_powers": float(np.mean(np.std(log_filled, axis=1))),
                "max_value_across_pairs": float(np.max(filled)),
                "positive_fraction_overall": float(np.mean(group["super_rate_cc_per_s"] > 0.0)),
            }
        )
    power_sensitivity = pd.DataFrame(power_sensitivity_rows).sort_values(
        "mean_log_std_across_powers", ascending=False
    )
    save_csv(power_sensitivity, args.output_dir / "super_rate_power_sensitivity_summary.csv")

    relation_rows = []
    for (reaction_id, reaction_label), group in rate_long.groupby(["reaction_id", "reaction_label"], sort=False):
        positive_mask = (group["super_rate_cc_per_s"] > 0.0) & (group["rate_const"] > 0.0)
        relation_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_label": reaction_label,
                "total_case_count": int(len(group)),
                "super_rate_positive_fraction": float(np.mean(group["super_rate_cc_per_s"] > 0.0)),
                "rate_const_positive_fraction": float(np.mean(group["rate_const"] > 0.0)),
                "joint_positive_fraction": float(np.mean(positive_mask)),
                "super_rate_given_rate_const_positive": float(
                    positive_mask.sum() / max(int((group["rate_const"] > 0.0).sum()), 1)
                ),
                "log_corr_with_rate_const_positive_only": safe_log_corr(
                    group["rate_const"], group["super_rate_cc_per_s"]
                ),
            }
        )
    relation_frame = pd.DataFrame(relation_rows).sort_values(
        "log_corr_with_rate_const_positive_only", ascending=False
    )
    save_csv(relation_frame, args.output_dir / "super_rate_rate_const_relationship_summary.csv")

    super_case_by_power = (
        case_summary.groupby(["power_label", "power_mj"], as_index=False)
        .agg(
            nonzero_super_rate_count_mean=("nonzero_super_rate_count", "mean"),
            super_rate_sum_mean=("super_rate_sum", "mean"),
            super_rate_max_mean=("super_rate_max", "mean"),
            super_rate_median_mean=("super_rate_median", "mean"),
        )
        .sort_values("power_mj")
    )
    save_csv(super_case_by_power, args.output_dir / "super_rate_case_by_power_summary.csv")

    manifest = pd.DataFrame(
        [
            {"path": str(args.output_dir / "super_rate_overall_summary.csv"), "description": "Overall SUPER RATE distribution summary."},
            {"path": str(args.output_dir / "super_rate_by_power_summary.csv"), "description": "SUPER RATE distribution summary by power."},
            {"path": str(args.output_dir / "super_rate_case_summary.csv"), "description": "Case-level SUPER RATE summary table."},
            {"path": str(args.output_dir / "super_rate_local_case_summary.csv"), "description": "Average SUPER RATE behavior by local E/N case."},
            {"path": str(args.output_dir / "super_rate_reaction_summary.csv"), "description": "Per-reaction SUPER RATE statistics."},
            {"path": str(args.output_dir / "super_rate_reaction_by_power_summary.csv"), "description": "Per-reaction SUPER RATE statistics by power."},
            {"path": str(args.output_dir / "super_rate_power_sensitivity_summary.csv"), "description": "Reaction-wise SUPER RATE sensitivity to power."},
            {"path": str(args.output_dir / "super_rate_rate_const_relationship_summary.csv"), "description": "Relationship between SUPER RATE and RATE CONST by reaction."},
            {"path": str(args.output_dir / "super_rate_case_by_power_summary.csv"), "description": "Case-level SUPER RATE aggregates by power."},
        ]
    )
    save_csv(manifest, args.output_dir / "super_rate_analysis_manifest.csv")
    print(f"Wrote super-rate analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
